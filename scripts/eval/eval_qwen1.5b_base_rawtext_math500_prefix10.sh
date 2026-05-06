#!/bin/bash
# Qwen2.5-Math-1.5B (Base), MATH-500, prefix, 10 tokens, raw-text prompt
python eval.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --data_dir eval/data --data_names math500 --split test \
    --prompt_type mathstral \
    --batch_size 32 --max_tokens_per_call 3072 \
    --temperature 0 --seed 0 \
    --embedding_mode prefix \
    --num_embedding_tokens 10 \
    --output_dir output/prefix_qwen1.5b_base_rawtext_tok10_math500
