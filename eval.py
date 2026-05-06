"""Unified evaluation script for RSP (Random Soft Prompt) experiments.

Supports:
- 4 embedding modes: none (baseline), prefix, middle, suffix
- Multiple models: Qwen2.5-Math, Qwen2.5 base/instruct, LLaMA
- Benchmarks: MATH-500, GSM8K
"""
import os
import sys
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Add eval/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

from utils import set_seed, load_jsonl, save_jsonl, construct_prompt, PROMPT_TEMPLATES
from parser import parse_question, parse_ground_truth, extract_answer, strip_string, run_execute
from data_loader import load_data
from python_executor import PythonExecutor
from grader import math_equal_process

from model_utils import (
    load_model_and_tokenizer,
    generate_completions,
    build_middle_prompt,
    split_prompt_for_suffix,
    get_special_tokens_str,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RSP Evaluation")

    # Data
    parser.add_argument("--data_names", default="math500", type=str,
                        help="Comma-separated dataset names (math500, gsm8k)")
    parser.add_argument("--data_dir", default="eval/data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)

    # Model
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str,
                        help="Prompt template from utils.py")
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Use tokenizer.apply_chat_template instead of PROMPT_TEMPLATES")

    # Embedding injection
    parser.add_argument("--embedding_mode", default="none", type=str,
                        choices=["none", "prefix", "middle", "suffix"],
                        help="Random embedding injection mode")
    parser.add_argument("--embedding_path", default="", type=str,
                        help="Path to directory containing prefix_embeddings.pt")
    parser.add_argument("--num_embedding_tokens", default=20, type=int,
                        help="Number of random embedding tokens")
    parser.add_argument("--suffix_position", default="before_im_end", type=str,
                        choices=["before_im_end", "between_turns", "after_assistant"],
                        help="Where to concat random embeddings in suffix mode")

    # Generation
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_tokens_per_call", default=3072, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--attn_implementation", default=None, type=str,
                        choices=[None, "eager", "sdpa", "flash_attention_2"],
                        help="Attention implementation (default: transformers default)")

    # Output (jsonl saved by default)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--no_save_outputs", action="store_false", dest="save_outputs",
                        help="Disable saving outputs to jsonl (default: enabled)")
    parser.set_defaults(save_outputs=True)
    parser.add_argument("--overwrite", action="store_true")

    # Wandb (optional)
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging (default: off)")
    parser.add_argument("--wandb_project", default="rsp", type=str,
                        help="Wandb project name (used only with --use_wandb)")
    parser.add_argument("--wandb_run_name", default=None, type=str)

    # Embedding seed
    parser.add_argument("--embedding_seed", default=15, type=int,
                        help="Base seed for random embedding generation (actual seed = embedding_seed + batch_start)")

    # Few-shot
    parser.add_argument("--num_shots", default=0, type=int)
    parser.add_argument("--adapt_few_shot", action="store_true")

    # Entropy recording
    parser.add_argument("--record_entropy", action="store_true",
                        help="Record per-step entropy, varentropy, top1_prob during generation")

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def prepare_data(data_name, args):
    """Load and prepare dataset."""
    examples = load_data(data_name, args.split, args.data_dir)

    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.embedding_mode}_seed{args.seed}"
    output_dir = args.output_dir
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # Resume support: load already processed samples
    processed_samples = []
    if not args.overwrite:
        processed_dir = f"{output_dir}/{data_name}/"
        if os.path.exists(processed_dir):
            processed_files = [
                f for f in os.listdir(processed_dir)
                if f.endswith(".jsonl") and f.startswith(out_file_prefix)
            ]
            for f in processed_files:
                processed_samples.extend(list(load_jsonl(f"{processed_dir}/{f}")))

    processed_samples = {s["idx"]: s for s in processed_samples}
    processed_idxs = set(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [e for e in examples if e["idx"] not in processed_idxs]

    return examples, processed_samples, out_file


def main_eval(llm, tokenizer, data_name, args, random_embeds, embed_stats=None):
    """Run evaluation on a single dataset."""
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print(f"Data: {data_name}, remaining samples: {len(examples)}")
    if len(examples) > 0:
        print(f"First example: {examples[0].get('problem', examples[0].get('question', ''))[:100]}...")

    executor = PythonExecutor(get_answer_from_stdout=True)

    # Build per-example prompts
    samples = []
    for example in tqdm(examples, desc="Preparing prompts"):
        idx = example["idx"]
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        if args.apply_chat_template:
            # Instruct models (LLaMA etc.): use tokenizer.apply_chat_template
            # with system + user messages.
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": example["question"]},
            ]
            if args.embedding_mode == "middle":
                special_str = get_special_tokens_str(args.model_name_or_path, args.num_embedding_tokens)
                description = (
                    f'\n\nThere are {args.num_embedding_tokens} special tokens that contain compressed latent reasoning information '
                    f'that might be useful for your reasoning.\n'
                    f'If these tokens are useful for your case, you can use them as reference. If these tokens are not useful '
                    f'for your case, you can ignore them and focus back to solving the problem.\n\n'
                    f'Here are the {args.num_embedding_tokens} special tokens: {special_str}'
                )
                messages[1]["content"] = example["question"] + description
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif args.embedding_mode == "middle":
            # Middle injection: description text + special tokens in prompt
            full_prompt = build_middle_prompt(
                example["question"], args.model_name_or_path,
                args.num_embedding_tokens, args.prompt_type
            )
        else:
            full_prompt = construct_prompt(example, data_name, args)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }
        for key in ["level", "type", "unit", "solution_type", "choices", "solution",
                     "ques_type", "ans_type", "answer_type", "dataset", "subfield",
                     "filed", "theorem", "answer", "difficulty"]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    if len(samples) == 0:
        print("No new samples to process.")
        if processed_samples:
            all_samples, result_json = evaluate(
                samples=processed_samples, data_name=data_name,
                prompt_type=args.prompt_type, execute=True,
            )
            return result_json
        return {"acc": 0}

    # Print first prompt for verification
    print(f"\n--- First prompt ({args.prompt_type}, mode={args.embedding_mode}) ---")
    print(samples[0]["prompt"][:500])
    print("---\n")

    # Stop words
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")

    # Batch inference
    start_time = time.time()
    all_samples = list(processed_samples)
    correct = 0
    total = 0
    batch_size = args.batch_size

    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        batch_prompts = [s["prompt"] for s in batch_samples for _ in range(args.n_sampling)]

        # Re-generate random embeddings per batch (deterministic via seed)
        if embed_stats is not None:
            torch.manual_seed(args.embedding_seed + batch_start)
            random_embeds = torch.randn(
                args.num_embedding_tokens, embed_stats["hidden_dim"]
            ) * embed_stats["std"] + embed_stats["mean"]

        # Suffix mode: split prompts into before/after parts
        batch_prompts_after = None
        if args.embedding_mode == "suffix":
            has_chat_markers = "<|im_end|>" in batch_prompts[0] or "<|eot_id|>" in batch_prompts[0]
            if has_chat_markers:
                # ChatML/chat template: split at marker position
                split_results = [
                    split_prompt_for_suffix(p, args.model_name_or_path, args.suffix_position)
                    for p in batch_prompts
                ]
                batch_prompts = [r[0] for r in split_results]
                batch_prompts_after = [r[1] for r in split_results]
            # else: no chat markers; concat at end (prompts_after stays None)

        # Generate
        gen_kwargs = {
            "max_new_tokens": args.max_tokens_per_call,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature if args.temperature > 0 else None,
            "top_p": args.top_p if args.temperature > 0 else None,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        gen_result = generate_completions(
            model=llm,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            random_embeds=random_embeds,
            embedding_mode=args.embedding_mode,
            model_name=args.model_name_or_path,
            num_tokens=args.num_embedding_tokens,
            suffix_position=args.suffix_position,
            batch_size=batch_size,
            stop_words=stop_words,
            prompts_after=batch_prompts_after,
            record_entropy=args.record_entropy,
            **gen_kwargs,
        )
        if args.record_entropy:
            outputs, batch_entropy_data = gen_result
        else:
            outputs = gen_result
            batch_entropy_data = None

        # Grade each sample
        for i, sample in enumerate(batch_samples):
            codes_i = []
            for j in range(args.n_sampling):
                out_idx = i * args.n_sampling + j
                raw_output = outputs[out_idx]
                # Strip the input prompt from the model output
                code = raw_output.split(batch_prompts[out_idx])[-1].strip() \
                    if batch_prompts[out_idx] in raw_output else raw_output.strip()
                for sw in stop_words:
                    if sw in code:
                        code = code.split(sw)[0].strip()
                codes_i.append(code)

            results_i = [run_execute(executor, c, args.prompt_type, data_name) for c in codes_i]
            preds = [item[0] for item in results_i]
            reports = [item[1] for item in results_i]

            gen_token_counts = [len(tokenizer.encode(c, add_special_tokens=False)) for c in codes_i]

            sample.pop("prompt")
            sample.update({"code": codes_i, "pred": preds, "report": reports, "gen_tokens": gen_token_counts})

            # Trim entropy curves to gen_tokens length when recording
            if batch_entropy_data is not None:
                out_idx = i * args.n_sampling + (args.n_sampling - 1)
                trim_len = gen_token_counts[-1]
                sample["entropy_curve"] = batch_entropy_data[out_idx]["entropy"][:trim_len]
                sample["varentropy_curve"] = batch_entropy_data[out_idx]["varentropy"][:trim_len]
                sample["top1_prob_curve"] = batch_entropy_data[out_idx]["top1_prob"][:trim_len]

            scores_i = [math_equal_process((0, p, sample["gt"])) for p in preds]
            sample["score"] = scores_i
            total += 1
            if scores_i[-1]:
                correct += 1

            all_samples.append(sample)

            if args.use_wandb and _WANDB_AVAILABLE:
                wandb.log({
                    f"problem/{data_name}_running_accuracy": correct / total,
                    f"problem/{data_name}_running_total": total,
                }, step=sample["idx"])

        acc_str = f"{correct / total:.4f}" if total > 0 else "N/A"
        print(f"[Batch {batch_start // batch_size + 1}] {total}/{len(samples)} done, "
              f"running acc: {acc_str}", flush=True)

        if args.save_outputs:
            save_jsonl(all_samples, out_file)

    time_use = time.time() - start_time

    final_acc = round(correct / total * 100, 2) if total > 0 else 0
    empty_count = sum(1 for s in all_samples if not (s['pred'][-1] if isinstance(s['pred'], list) else s['pred']))

    # Per-type accuracy
    type_acc = {}
    if all_samples and "type" in all_samples[0]:
        type_scores = {}
        for s in all_samples:
            t = s.get("type", "unknown")
            if t not in type_scores:
                type_scores[t] = []
            sc = s["score"][-1] if isinstance(s["score"], list) else s["score"]
            type_scores[t].append(sc)
        type_acc = {k: round(sum(v) / len(v) * 100, 2) for k, v in sorted(type_scores.items())}

    # Mean generated tokens
    all_gen_tokens = []
    for s in all_samples:
        if "gen_tokens" in s:
            gt = s["gen_tokens"]
            all_gen_tokens.extend(gt if isinstance(gt, list) else [gt])
    mean_gen_tokens = round(sum(all_gen_tokens) / len(all_gen_tokens), 1) if all_gen_tokens else 0

    result_json = {
        "num_samples": total,
        "acc": final_acc,
        "correct": correct,
        "empty_samples": empty_count,
        "mean_gen_tokens": mean_gen_tokens,
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
    }
    if type_acc:
        result_json["type_acc"] = type_acc

    if args.save_outputs:
        save_jsonl(all_samples, out_file)

    metrics_file = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(result_json, f, indent=4)

    return result_json


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load model
    print(f"Loading model: {args.model_name_or_path}")
    llm, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
    )

    # Random embedding stats (actual embeddings generated per batch)
    random_embeds = None
    embed_stats = None
    if args.embedding_mode != "none":
        if args.embedding_path:
            embed_path = Path(args.embedding_path) / "prefix_embeddings.pt"
            print(f"Loading random embeddings from {embed_path}")
            state_dict = torch.load(embed_path, map_location="cpu")
            random_embeds = state_dict["weight"]
            print(f"Loaded embeddings shape: {random_embeds.shape}")
        else:
            embed_weight = llm.get_input_embeddings().weight.data.cpu()
            embed_stats = {
                "mean": embed_weight.mean(),
                "std": embed_weight.std(),
                "hidden_dim": embed_weight.shape[1],
            }
            print(f"Random embedding mode: {args.embedding_mode}, tokens={args.num_embedding_tokens}, regenerate per batch")

    # Wandb init (optional)
    if args.use_wandb and _WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Run evaluation per dataset
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        result = main_eval(llm, tokenizer, data_name, args, random_embeds, embed_stats)
        results.append(result)

    # Cross-dataset average
    data_list.append("avg")
    results.append({"acc": sum(r["acc"] for r in results) / len(results)})

    # Final wandb log (optional)
    if args.use_wandb and _WANDB_AVAILABLE:
        log_dict = {}
        for data_name, result in zip(data_list, results):
            log_dict[f"acc/{data_name}"] = result["acc"]
            if "type_acc" in result:
                for type_name, type_acc in result["type_acc"].items():
                    log_dict[f"type_acc/{data_name}/{type_name}"] = type_acc
        wandb.log(log_dict)
        wandb.finish()

    # Print final results
    pad = max(len(d) for d in data_list)
    print("\t".join(d.ljust(pad) for d in data_list))
    print("\t".join(f"{r['acc']:.1f}".ljust(pad) for r in results))


if __name__ == "__main__":
    main()
