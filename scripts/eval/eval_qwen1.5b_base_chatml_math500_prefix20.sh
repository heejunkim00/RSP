#!/bin/bash
# Qwen2.5-Math-1.5B (Base), MATH-500, prefix, 20 tokens, chatml prompt
python eval.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --data_dir eval/data --data_names math500 --split test \
    --prompt_type qwen25-math-cot \
    --batch_size 32 --max_tokens_per_call 3072 \
    --temperature 0 --seed 0 \
    --embedding_mode prefix \
    --num_embedding_tokens 20 \
    --output_dir output/prefix_qwen1.5b_base_chatml_tok20_math500
