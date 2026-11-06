"""Evaluate a causal LM on math benchmarks with or without a Random Soft Prompt (RSP).

One invocation = one (model, benchmark(s), injection mode, L, RSP seed) cell.

Embedding modes
    none    : baseline
    prefix  : [H ; X]
    suffix  : [X ; H] inserted at --suffix_position (paper default: before_im_end)

The RSP H is redrawn for every batch from N(mu_E, sigma_E^2 I) with
torch seed = --embedding_seed + batch_start, so a run is fully determined by
(--embedding_seed, --batch_size). Keep --batch_size fixed per model when
comparing to the paper (padding changes greedy outputs).
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))

from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import parse_question, parse_ground_truth, run_execute
from data_loader import load_data
from python_executor import PythonExecutor
from grader import math_equal_process

from model_utils import (
    load_model_and_tokenizer,
    generate_completions,
    split_prompt_for_suffix,
    embedding_table_stats,
    sample_rsp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RSP evaluation")

    # Data
    parser.add_argument("--data_names", default="math500", type=str,
                        help="Comma-separated: math500, gsm8k, aime24")
    parser.add_argument("--data_dir", default="eval/data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int,
                        help="Use only the first N problems (smoke tests)")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)

    # Model / prompt
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str,
                        help="Template in eval/utils.py (qwen25-math-cot | mathstral)")
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Wrap the question with tokenizer.apply_chat_template "
                             "(used for LLaMA-3.1-8B-Instruct with --prompt_type mathstral)")
    parser.add_argument("--attn_implementation", default=None, type=str,
                        choices=[None, "eager", "sdpa", "flash_attention_2"])

    # RSP
    parser.add_argument("--embedding_mode", default="none", type=str,
                        choices=["none", "prefix", "suffix"])
    parser.add_argument("--num_embedding_tokens", default=20, type=int,
                        help="RSP length L")
    parser.add_argument("--embedding_seed", default=15, type=int,
                        help="Base seed; batch b uses seed embedding_seed + b*batch_size")
    parser.add_argument("--suffix_position", default="before_im_end", type=str,
                        choices=["before_im_end", "between_turns", "after_assistant"])
    parser.add_argument("--embedding_path", default="", type=str,
                        help="Optional: load a fixed H from <dir>/prefix_embeddings.pt "
                             "(see generate_random_embedding.py) instead of resampling")

    # Decoding
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_tokens_per_call", default=3072, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int, help="Decoding RNG seed")

    # Output
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing jsonl in output_dir instead of resuming")

    # Optional wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rsp", type=str)
    parser.add_argument("--wandb_run_name", default=None, type=str)

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.embedding_mode}_seed{args.seed}"
    out_file = f"{args.output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{args.output_dir}/{data_name}", exist_ok=True)

    # Resume: skip problems already present in output_dir
    processed_samples = []
    if not args.overwrite:
        processed_dir = f"{args.output_dir}/{data_name}/"
        if os.path.exists(processed_dir):
            for f in os.listdir(processed_dir):
                if f.endswith(".jsonl") and f.startswith(out_file_prefix):
                    processed_samples.extend(list(load_jsonl(f"{processed_dir}/{f}")))
    processed_samples = {s["idx"]: s for s in processed_samples}
    processed_idxs = set(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [e for e in examples if e["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def build_prompt(example, data_name, tokenizer, args):
    if args.apply_chat_template:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": example["question"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return construct_prompt(example["question"], args.prompt_type)


def main_eval(llm, tokenizer, data_name, args, fixed_embeds, embed_stats):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print(f"Data: {data_name}, remaining problems: {len(examples)}")

    executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, desc="Preparing prompts"):
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        sample = {
            "idx": example["idx"],
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": build_prompt(example, data_name, tokenizer, args),
        }
        for key in ["level", "type", "answer", "solution", "url"]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    all_samples = list(processed_samples)
    if samples:
        print(f"\n--- First prompt ({args.prompt_type}, mode={args.embedding_mode}) ---")
        print(samples[0]["prompt"][:500])
        print("---\n")

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
    gen_kwargs = {"max_new_tokens": args.max_tokens_per_call, "do_sample": args.temperature > 0}
    if args.temperature > 0:
        gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    start_time = time.time()
    correct = total = 0
    batch_size = args.batch_size

    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        batch_prompts = [s["prompt"] for s in batch_samples for _ in range(args.n_sampling)]

        # Fresh RSP draw per batch (Eq. 1), deterministic in the seed
        random_embeds = fixed_embeds
        if embed_stats is not None:
            random_embeds = sample_rsp(args.num_embedding_tokens, embed_stats,
                                       args.embedding_seed + batch_start)

        # Suffix mode on chat-formatted prompts: split at the insertion point
        batch_prompts_after = None
        if args.embedding_mode == "suffix" and (
            "<|im_end|>" in batch_prompts[0] or "<|eot_id|>" in batch_prompts[0]
        ):
            split_results = [
                split_prompt_for_suffix(p, args.model_name_or_path, args.suffix_position)
                for p in batch_prompts
            ]
            batch_prompts = [r[0] for r in split_results]
            batch_prompts_after = [r[1] for r in split_results]

        outputs = generate_completions(
            model=llm, tokenizer=tokenizer, prompts=batch_prompts,
            random_embeds=random_embeds, embedding_mode=args.embedding_mode,
            num_tokens=args.num_embedding_tokens, batch_size=batch_size,
            stop_words=stop_words, prompts_after=batch_prompts_after, **gen_kwargs,
        )

        for i, sample in enumerate(batch_samples):
            codes_i = []
            for j in range(args.n_sampling):
                code = outputs[i * args.n_sampling + j].strip()
                for sw in stop_words:
                    if sw in code:
                        code = code.split(sw)[0].strip()
                codes_i.append(code)

            results_i = [run_execute(executor, c, args.prompt_type, data_name) for c in codes_i]
            preds = [r[0] for r in results_i]
            scores_i = [math_equal_process((0, p, sample["gt"])) for p in preds]

            sample.pop("prompt")
            sample.update({
                "code": codes_i,
                "pred": preds,
                "score": scores_i,
                "gen_tokens": [len(tokenizer.encode(c, add_special_tokens=False)) for c in codes_i],
            })
            total += 1
            correct += int(scores_i[-1])
            all_samples.append(sample)

            if args.use_wandb and _WANDB_AVAILABLE:
                wandb.log({f"problem/{data_name}_running_accuracy": correct / total}, step=sample["idx"])

        acc_str = f"{correct / total:.4f}" if total > 0 else "N/A"
        print(f"[Batch {batch_start // batch_size + 1}] {total}/{len(samples)} done, "
              f"running acc: {acc_str}", flush=True)
        save_jsonl(all_samples, out_file)

    # Metrics over everything in the output file (resumed + new)
    all_samples.sort(key=lambda s: s["idx"])
    save_jsonl(all_samples, out_file)
    final_scores = [s["score"][-1] for s in all_samples]
    n = len(final_scores)
    result_json = {
        "num_samples": n,
        "acc": round(sum(final_scores) / n * 100, 2) if n else 0.0,
        "correct": int(sum(final_scores)),
        "n_sampling": args.n_sampling,
        "embedding_mode": args.embedding_mode,
        "num_embedding_tokens": args.num_embedding_tokens if args.embedding_mode != "none" else 0,
        "embedding_seed": args.embedding_seed if args.embedding_mode != "none" else None,
        "time_use_in_second": round(time.time() - start_time, 1),
    }
    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading model: {args.model_name_or_path}")
    llm, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, attn_implementation=args.attn_implementation
    )

    fixed_embeds, embed_stats = None, None
    if args.embedding_mode != "none":
        if args.embedding_path:
            embed_path = Path(args.embedding_path)
            if embed_path.is_dir():
                embed_path = embed_path / "prefix_embeddings.pt"
            fixed_embeds = torch.load(embed_path, map_location="cpu")["weight"]
            assert fixed_embeds.shape[0] == args.num_embedding_tokens, \
                f"--num_embedding_tokens={args.num_embedding_tokens} but file has {fixed_embeds.shape[0]} rows"
            print(f"Loaded fixed RSP from {embed_path}: {tuple(fixed_embeds.shape)}")
        else:
            embed_stats = embedding_table_stats(llm)
            print(f"RSP mode={args.embedding_mode}, L={args.num_embedding_tokens}, "
                  f"mu_E={embed_stats['mean']:.3e}, sigma_E={embed_stats['std']:.3e}, "
                  f"redrawn per batch from seed {args.embedding_seed}")

    if args.use_wandb and _WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    data_list = args.data_names.split(",")
    results = [main_eval(llm, tokenizer, d, args, fixed_embeds, embed_stats) for d in data_list]

    if args.use_wandb and _WANDB_AVAILABLE:
        wandb.log({f"acc/{d}": r["acc"] for d, r in zip(data_list, results)})
        wandb.finish()

    pad = max(len(d) for d in data_list)
    print("\t".join(d.ljust(pad) for d in data_list))
    print("\t".join(f"{r['acc']:.2f}".ljust(pad) for r in results))


if __name__ == "__main__":
    main()
