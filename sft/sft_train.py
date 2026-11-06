"""Supervised fine-tuning on collected (problem, response) pairs with FSDP.

    torchrun --nproc_per_node=8 sft/sft_train.py \
        --data out/rsp_seed42/correct_rsp.jsonl --model Qwen/Qwen2.5-Math-7B-Instruct \
        --output_dir out/rsp_seed42/sft --grad_accum 2

Prompt = the same chat prompt used for collection (no RSP). Loss is on
response tokens only. Effective batch = batch_size * grad_accum * #GPUs.
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.tok = tokenizer
        self.max_length = max_length
        with open(path) as f:
            self.data = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(item["problem"])}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
        resp_ids = self.tok.encode(item["response"], add_special_tokens=False)
        if self.tok.eos_token_id is not None and (not resp_ids or resp_ids[-1] != self.tok.eos_token_id):
            resp_ids.append(self.tok.eos_token_id)
        # Keep the response; truncate the prompt from the left if needed
        if len(prompt_ids) + len(resp_ids) > self.max_length:
            resp_ids = resp_ids[: self.max_length]
            prompt_ids = prompt_ids[len(prompt_ids) - max(self.max_length - len(resp_ids), 0):]
        return {
            "input_ids": prompt_ids + resp_ids,
            "labels": [-100] * len(prompt_ids) + resp_ids,
        }


class LeftPadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: [v] * (n - len(seq)) + seq
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl with problem / answer / response")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=2, help="per device")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help="use 'sdpa' if flash-attn is not installed")
    ap.add_argument("--max_steps", type=int, default=-1, help="override epochs (smoke test)")
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()

    rank0 = int(os.environ.get("RANK", 0)) == 0
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = SFTDataset(args.data, tokenizer, args.max_length)
    if rank0:
        print(f"[SFT] {len(dataset)} examples from {args.data}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    layer_cls = type(model.model.layers[0]).__name__

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        max_grad_norm=1.0,
        seed=args.seed,
        logging_steps=5,
        save_strategy="no",
        report_to=["wandb"] if args.wandb_project else [],
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": [layer_cls], "use_orig_params": True},
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=LeftPadCollator(tokenizer.pad_token_id), tokenizer=tokenizer,
    )
    trainer.train()

    # Gather the full state dict and save a materialized copy of every tensor.
    state_dict = trainer.accelerator.get_state_dict(trainer.model)
    if rank0 and state_dict is not None:
        state_dict = {k: v.detach().to("cpu", copy=True).contiguous().clone() for k, v in state_dict.items()}
        trainer.accelerator.unwrap_model(trainer.model).save_pretrained(
            args.output_dir, state_dict=state_dict, safe_serialization=True
        )
        tokenizer.save_pretrained(args.output_dir)
        print(f"[SFT] saved to {args.output_dir}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
