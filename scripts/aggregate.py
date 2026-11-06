"""Aggregate scripts/main_tables.sh outputs into mean +- std over RSP seeds.

Usage:
    python scripts/aggregate.py output/main_tables [--markdown]

Expects the layout written by main_tables.sh:
    <root>/<config>/baseline/<dataset>/*_metrics.json
    <root>/<config>/<mode>_L<L>_seed<seed>/<dataset>/*_metrics.json
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = ["math500", "gsm8k", "aime24"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    # (config, condition, dataset) -> list of acc
    table = defaultdict(list)
    for mf in sorted(args.root.glob("*/*/*/*_metrics.json")):
        config, run, dataset = mf.parts[-4], mf.parts[-3], mf.parts[-2]
        acc = json.load(open(mf))["acc"]
        m = re.match(r"(prefix|suffix)_L(\d+)_seed(\d+)", run)
        cond = f"{m.group(1)} (L={m.group(2)})" if m else run
        table[(config, cond, dataset)].append(acc)

    configs = sorted({k[0] for k in table})
    conds = sorted({k[1] for k in table}, key=lambda c: (c != "baseline", c))

    def cell(v):
        if not v:
            return "-"
        if len(v) == 1:
            return f"{v[0]:.2f}"
        return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f} (n={len(v)})"

    sep = " | " if args.markdown else "\t"
    header = ["config", "condition"] + DATASETS
    print(sep.join(header))
    if args.markdown:
        print(sep.join(["---"] * len(header)))
    for c in configs:
        for cond in conds:
            row = [cell(table.get((c, cond, d), [])) for d in DATASETS]
            if all(r == "-" for r in row):
                continue
            print(sep.join([c, cond] + row))


if __name__ == "__main__":
    main()
