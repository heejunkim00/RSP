"""Evaluate an SFT checkpoint with vLLM.

    # greedy benchmarks
    python sft/eval_sft.py --model <ckpt> --data_names minerva_math,gaokao2023en,olympiadbench
    # Avg@32 benchmarks
    python sft/eval_sft.py --model <ckpt> --data_names aime24,amc23 --temperature 1.0 --top_p 0.95 --n_sampling 32

Benchmark files come from `python sft/download_eval_data.py` (sft/eval_data/<name>/test.jsonl).
Reported `acc` is the mean over all (problem, sample) pairs (Avg@n). The answer is the
last \\boxed{} of the response; equivalence checking reuses eval/grader.py.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import TimeoutError

from pebble import ProcessPool
from vllm import LLM, SamplingParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
from utils import set_seed, save_jsonl, PROMPT_TEMPLATES  # noqa: E402
from data_loader import load_data  # noqa: E402
from parser import parse_question, parse_ground_truth, strip_string, STRIP_EXCEPTIONS  # noqa: E402
from grader import math_equal_process  # noqa: E402

PROMPT_TYPE = "qwen25-math-cot"
STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>", "assistant", "user", "_end", "_start"]


def last_boxed(text):
    ans = text.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] != "{":
        return ans.split("$")[0].strip()
    depth, out = 1, ""
    for c in ans[1:]:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out += c
    return out


def extract_answer(text, data_name):
    if "final answer is $" in text and "$. I hope" in text:
        pred = text.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
    elif "boxed" in text:
        pred = last_boxed(text)
    elif "he answer is" in text:
        pred = text.split("he answer is")[-1].strip()
    elif "final answer is" in text:
        pred = text.split("final answer is")[-1].strip()
    else:
        nums = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
        pred = nums[-1] if nums else ""
    pred = re.sub(r"\n\s*", "", pred).strip(":").rstrip("./")
    return strip_string(pred, skip_unit=data_name in STRIP_EXCEPTIONS)


def grade(params, timeout=3):
    """params: list of (idx, pred, gt). Returns list of bool; timeouts count as wrong."""
    scores = []
    with ProcessPool(max_workers=1) as pool:
        it = pool.map(math_equal_process, params, timeout=timeout).result()
        while True:
            try:
                scores.append(bool(next(it)))
            except StopIteration:
                break
            except TimeoutError:
                scores.append(False)
    return scores


def evaluate_dataset(llm, tokenizer, data_name, args):
    examples = load_data(data_name, "test", args.data_dir)
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    samples = []
    for ex in examples:
        ex["question"] = parse_question(ex, data_name)
        gt_cot, gt = parse_ground_truth(ex, data_name)
        samples.append({"idx": ex["idx"], "question": ex["question"], "gt": gt,
                        "prompt": PROMPT_TEMPLATES[PROMPT_TYPE].format(input=ex["question"]).strip(" ")})

    stop_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, n=args.n_sampling,
                        max_tokens=args.max_tokens, stop=STOP_WORDS,
                        stop_token_ids=[i for i in stop_ids if i is not None])
    t0 = time.time()
    outputs = llm.generate([s["prompt"] for s in samples], sp)

    params = []
    for s, out in zip(samples, outputs):
        s["code"] = [o.text.strip() for o in out.outputs]
        s["pred"] = [extract_answer(c, data_name) for c in s["code"]]
        params += [(s["idx"], p, s["gt"]) for p in s["pred"]]
        s.pop("prompt")
    scores = grade(params)
    k = 0
    for s in samples:
        s["score"] = scores[k:k + len(s["pred"])]
        k += len(s["pred"])

    acc = 100 * sum(scores) / len(scores)
    out_dir = os.path.join(args.output_dir, data_name)
    save_jsonl(samples, os.path.join(out_dir, f"test_t{args.temperature}_n{args.n_sampling}.jsonl"))
    metrics = {"num_samples": len(samples), "n_sampling": args.n_sampling, "temperature": args.temperature,
               "acc": round(acc, 2), "time_sec": round(time.time() - t0)}
    with open(os.path.join(out_dir, f"metrics_t{args.temperature}_n{args.n_sampling}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_names", default="minerva_math,gaokao2023en,olympiadbench")
    ap.add_argument("--data_dir", default="sft/eval_data")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--n_sampling", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--num_test_sample", type=int, default=-1)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.top_p = 1.0 if args.temperature == 0 else args.top_p
    set_seed(args.seed)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp_size, dtype="bfloat16", seed=args.seed)
    tokenizer = llm.get_tokenizer()

    names = args.data_names.split(",")
    results = [evaluate_dataset(llm, tokenizer, d, args) for d in names]
    pad = max(len(d) for d in names)
    print("\t".join(d.ljust(pad) for d in names))
    print("\t".join(f"{r['acc']:.2f}".ljust(pad) for r in results))


if __name__ == "__main__":
    main()
