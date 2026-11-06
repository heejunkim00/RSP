"""Compute Pass@k curves from scripts/pass_at_n.sh outputs.

Usage:
    python scripts/pass_at_n.py output/pass_at_n/math500 [--plot out.pdf] [--max_k 16]

Each condition directory may contain one run with n_sampling=N (score list of
length N per problem) or several runs with n_sampling=1 (one score per problem
per run, e.g. eseed0..eseed15).  Scores are concatenated per problem idx, and
Pass@k uses the unbiased estimator of Chen et al. (2021):

    pass@k = 1 - C(n - c, k) / C(n, k)
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

CONDITIONS = [
    ("baseline", "Baseline (τ=1.0)"),
    ("rsp_single_seed", "RSP single seed (τ=1.0)"),
    ("rsp_indep_greedy", "RSP indep. seed (greedy)"),
    ("rsp_indep_temp", "RSP indep. seed (τ=1.0)"),
]


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_condition(cond_dir):
    """Return {problem_idx: [0/1, ...]} concatenated over every jsonl under cond_dir."""
    scores = defaultdict(list)
    files = sorted(cond_dir.rglob("*.jsonl"))
    for f in files:
        for line in open(f):
            s = json.loads(line)
            scores[s["idx"]].extend(int(x) for x in s["score"])
    return scores, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="e.g. output/pass_at_n/math500")
    ap.add_argument("--max_k", type=int, default=16)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    loaded = {}
    for key, label in CONDITIONS:
        d = args.root / key
        if not d.exists():
            continue
        scores, nfiles = load_condition(d)
        n_per_problem = {len(v) for v in scores.values()}
        if len(n_per_problem) != 1:
            print(f"[warn] {key}: unequal sample counts per problem {sorted(n_per_problem)}")
        loaded[label] = scores
        print(f"{label}: {len(scores)} problems, {min(n_per_problem)} samples/problem, {nfiles} file(s)")

    max_k = min([args.max_k] + [min(len(v) for v in s.values()) for s in loaded.values()])
    ks = list(range(1, max_k + 1))
    curves = {
        label: [100 * sum(pass_at_k(len(v), sum(v), k) for v in scores.values()) / len(scores) for k in ks]
        for label, scores in loaded.items()
    }

    print("\nk\t" + "\t".join(curves))
    for i, k in enumerate(ks):
        print(f"{k}\t" + "\t".join(f"{curves[c][i]:.2f}" for c in curves))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        for label, curve in curves.items():
            plt.plot(ks, curve, marker="o", markersize=3, label=label)
        plt.xlabel("N"); plt.ylabel("Pass@N (%)"); plt.xticks(ks)
        plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(args.plot)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
