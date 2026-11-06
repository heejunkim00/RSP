"""Rollout collection for rejection-sampling SFT (vLLM).

Two stages, both using the chat prompt
    system: "Please reason step by step, and put your final answer within \\boxed{}."
    user  : <problem>

1. Hard-set extraction: n clean rollouts per problem; keep problems with
   at most --hard_cutoff correct rollouts.

       python sft/collect_rollout.py extract --model <m> --problems sft/data/openr1_math_46k.parquet \
           --output_dir out/hard --n 4

2. Collection: n rollouts per hard problem, keep only correct ones.
   With --rsp, each of the n rollouts gets its own RSP (suffix, L tokens) that
   is injected through vLLM: the RSP vectors are written into unused rows of the
   embedding table and referenced by their token ids inside the prompt.

       python sft/collect_rollout.py collect --model <m> --hard_set out/hard/hard_set.jsonl \
           --output_dir out/rsp_seed42 --n 8 --rsp --seed 42

--shard i/k processes every k-th problem (for running one process per GPU);
outputs of the shards can simply be concatenated.
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model() ships a closure to workers

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout_grader import is_correct  # noqa: E402

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def prompt_token_ids(problem, tokenizer):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(problem)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def insert_rsp_ids(ids, rsp_ids, tokenizer):
    """Suffix position: right before `<|im_end|>\\n<|im_start|>assistant\\n`."""
    end = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    for i in range(len(ids) - len(end), -1, -1):
        if ids[i:i + len(end)] == end:
            return ids[:i] + rsp_ids + ids[i:]
    raise ValueError("assistant turn marker not found in prompt")


# ---------------------------------------------------------------------------
# RSP injection into vLLM
# ---------------------------------------------------------------------------
def sample_rsp(num_vectors, hidden_dim, seed, std):
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randn(num_vectors, hidden_dim, generator=gen) * std


def _write_rows(model, rows_cpu, start):
    """Runs inside each vLLM worker; copies the shard of `rows_cpu` this rank owns."""
    inner = model.unwrap() if hasattr(model, "unwrap") else model
    embed = inner.model.embed_tokens
    weight = embed.weight
    if hasattr(embed, "shard_indices"):
        lo, hi = embed.shard_indices.org_vocab_start_index, embed.shard_indices.org_vocab_end_index
    else:
        lo, hi = 0, weight.shape[0]
    a, b = max(start, lo), min(start + rows_cpu.shape[0], hi)
    if a >= b:
        return False
    with torch.no_grad():
        weight.data[a - lo:b - lo].copy_(rows_cpu[a - start:b - start].to(weight.device, weight.dtype))
    return bool(torch.allclose(weight.data[a - lo, :5].float().cpu(), rows_cpu[a - start, :5].float(), atol=1e-3))


def write_rsp_into_vllm(llm, rsp, start):
    import functools
    results = llm.apply_model(functools.partial(_write_rows, rows_cpu=rsp.cpu().contiguous(), start=start))
    assert any(results), "RSP rows were not written to any worker"
    print(f"[RSP] wrote {rsp.shape[0]} vectors into embedding rows [{start}, {start + rsp.shape[0]})")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_problems(args, tokenizer):
    if args.command == "extract":
        df = pd.read_parquet(args.problems)
        items = df[["problem", "answer", "data_source"]].to_dict("records")
    else:
        items = read_jsonl(args.hard_set)
    if args.shard:
        i, k = map(int, args.shard.split("/"))
        items = items[i::k]
    n_before = len(items)
    items = [it for it in items if len(prompt_token_ids(it["problem"], tokenizer)) <= args.max_prompt_length]
    print(f"[data] {len(items)} problems ({n_before - len(items)} dropped: prompt > {args.max_prompt_length} tokens)")
    return items


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "collect"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", help="extract: parquet with problem/answer/data_source")
    ap.add_argument("--hard_set", help="collect: hard_set.jsonl(.gz)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n", type=int, default=8, help="rollouts per problem")
    ap.add_argument("--hard_cutoff", type=int, default=1, help="extract: hard if #correct <= cutoff")
    ap.add_argument("--rsp", action="store_true", help="collect: a fresh suffix RSP per rollout")
    ap.add_argument("--rsp_tokens", type=int, default=20, help="RSP length L")
    ap.add_argument("--rsp_std", type=float, default=0.0226, help="RSP entries ~ N(0, std^2)")
    ap.add_argument("--rsp_row_start", type=int, default=None,
                    help="first unused embedding row to hold RSP vectors (default: len(tokenizer))")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed, and RSP seed with --rsp")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--max_response_length", type=int, default=3052)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    ap.add_argument("--shard", default=None, help="i/k: process problems i, i+k, i+2k, ...")
    args = ap.parse_args()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    items = load_problems(args, tokenizer)

    rsp_rows = args.n * args.rsp_tokens if (args.command == "collect" and args.rsp) else 0
    if rsp_rows:
        row_start = args.rsp_row_start if args.rsp_row_start is not None else len(tokenizer)
        assert row_start >= len(tokenizer) and row_start + rsp_rows <= config.vocab_size, \
            f"need {rsp_rows} unused embedding rows in [{len(tokenizer)}, {config.vocab_size})"

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=args.max_prompt_length + args.max_response_length + (args.rsp_tokens if rsp_rows else 0),
        enable_prefix_caching=not rsp_rows,
        seed=args.seed,
    )

    if rsp_rows:
        rsp = sample_rsp(rsp_rows, config.hidden_size, args.seed, args.rsp_std)
        write_rsp_into_vllm(llm, rsp, row_start)
        # one prompt per (problem, rollout slot); slot j uses rows [start + j*L, start + (j+1)*L)
        prompts, owner = [], []
        for pi, it in enumerate(items):
            ids = prompt_token_ids(it["problem"], tokenizer)
            for j in range(args.n):
                rows = list(range(row_start + j * args.rsp_tokens, row_start + (j + 1) * args.rsp_tokens))
                prompts.append({"prompt_token_ids": insert_rsp_ids(ids, rows, tokenizer)})
                owner.append(pi)
        # top-k keeps the RSP rows (tied lm_head) from being sampled as output tokens
        sp = SamplingParams(temperature=args.temperature, top_p=1.0, top_k=50, n=1,
                            max_tokens=args.max_response_length)
    else:
        prompts = [{"prompt_token_ids": prompt_token_ids(it["problem"], tokenizer)} for it in items]
        owner = list(range(len(items)))
        sp = SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1, n=args.n,
                            max_tokens=args.max_response_length)

    outputs = llm.generate(prompts, sampling_params=sp)

    # Grade: per problem, list of (response, correct)
    graded = [[] for _ in items]
    for out, pi in zip(outputs, owner):
        for s in out.outputs:
            text = tokenizer.decode(s.token_ids, skip_special_tokens=True)
            graded[pi].append((text, is_correct(text, items[pi]["answer"])))

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f".shard{args.shard.replace('/', 'of')}" if args.shard else ""
    if args.command == "extract":
        hard = [{**it, "num_correct": sum(c for _, c in g)}
                for it, g in zip(items, graded) if sum(c for _, c in g) <= args.hard_cutoff]
        write_jsonl(hard, f"{args.output_dir}/hard_set{suffix}.jsonl")
        stats = {"problems": len(items), "hard": len(hard), "hard_ratio": len(hard) / max(len(items), 1)}
    else:
        correct = [{"problem": it["problem"], "answer": it["answer"], "response": text}
                   for it, g in zip(items, graded) for text, c in g if c]
        name = "correct_rsp" if args.rsp else "correct_baseline"
        write_jsonl(correct, f"{args.output_dir}/{name}{suffix}.jsonl")
        solved = sum(1 for g in graded if any(c for _, c in g))
        stats = {
            "problems": len(items),
            "rollouts": len(items) * args.n,
            "correct_rollouts": len(correct),
            "rollout_accuracy": len(correct) / max(len(items) * args.n, 1),
            "problem_coverage": solved / max(len(items), 1),
        }
    stats.update({"time_sec": round(time.time() - t0), "args": vars(args)})
    with open(f"{args.output_dir}/stats{suffix}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
