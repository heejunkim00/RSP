#!/bin/bash
# Qwen2.5-Math-1.5B-Instruct, MATH-500, baseline, with per-step entropy logging
python eval.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --data_dir eval/data --data_names math500 --split test \
    --prompt_type qwen25-math-cot \
    --batch_size 8 --max_tokens_per_call 3072 \
    --temperature 0 --seed 0 \
    --embedding_mode none \
    --record_entropy \
    --attn_implementation eager \
    --output_dir output/entropy_baseline_qwen1.5b_inst_math500
