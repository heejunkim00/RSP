"""Download evaluation benchmark jsonl files (5 benchmarks) used by the
math_eval pipeline.

Source: https://github.com/hkust-nlp/simpleRL-reason
        examples/simplelr_math_eval/data/<benchmark>/test.jsonl

Benchmarks downloaded:
    - GSM8K
    - MATH-500
    - College Math
    - Minerva Math
    - AIME 2024

Output:
    eval/simplerl_math_eval/data/<benchmark>/test.jsonl
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


REPO_RAW = "https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/master"
SUBPATH = "examples/simplelr_math_eval/data"

BENCHMARKS = [
    "gsm8k",
    "math500",
    "college_math",
    "minerva_math",
    "aime24",
]


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as resp:
        out_path.write_bytes(resp.read())
    size_kb = out_path.stat().st_size / 1024
    print(f"   -> {out_path} ({size_kb:.1f} KB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simplerl_math_eval" / "data",
        help="Output dir (default: eval/simplerl_math_eval/data)",
    )
    args = parser.parse_args(argv)

    print(f"Output dir: {args.out_dir}")
    for bench in BENCHMARKS:
        url = f"{REPO_RAW}/{SUBPATH}/{bench}/test.jsonl"
        out = args.out_dir / bench / "test.jsonl"
        download(url, out)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
