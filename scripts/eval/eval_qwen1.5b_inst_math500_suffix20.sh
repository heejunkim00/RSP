#!/bin/bash
# Qwen2.5-Math-1.5B-Instruct, MATH-500, suffix, 20 tokens
python eval.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --data_dir eval/data --data_names math500 --split test \
    --prompt_type qwen25-math-cot \
    --batch_size 32 --max_tokens_per_call 3072 \
    --temperature 0 --seed 0 \
    --embedding_mode suffix --suffix_position before_im_end \
    --num_embedding_tokens 20 \
    --output_dir output/suffix_qwen1.5b_inst_tok20_math500
