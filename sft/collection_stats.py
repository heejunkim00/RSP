"""Problem coverage and rollout accuracy of a collection run (Table: correct-solution collection).

    python sft/collection_stats.py --hard_set sft/data/hard_set_qwen25_math_7b_inst.jsonl.gz \
        --correct out/rsp_seed42/correct_rsp.jsonl --n 8
"""
import argparse
import gzip
import json


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard_set", required=True)
    ap.add_argument("--correct", required=True, help="correct_*.jsonl (concatenate shards first)")
    ap.add_argument("--n", type=int, default=8, help="rollouts per problem")
    args = ap.parse_args()

    hard = read_jsonl(args.hard_set)
    correct = read_jsonl(args.correct)
    solved = {c["problem"] for c in correct}
    print(f"hard problems      : {len(hard)}")
    print(f"correct rollouts   : {len(correct)}")
    print(f"problem coverage   : {100 * len(solved) / len(hard):.2f}%")
    print(f"rollout accuracy   : {100 * len(correct) / (len(hard) * args.n):.2f}%")


if __name__ == "__main__":
    main()
