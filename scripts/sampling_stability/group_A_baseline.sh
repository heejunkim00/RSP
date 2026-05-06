#!/bin/bash
# Group A: temperature diversity only (baseline, no softprompt)
#   16 samples per problem at temperature=1.0
python eval.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --data_dir eval/data --data_names math500 --split test \
    --prompt_type qwen25-math-cot \
    --batch_size 32 --max_tokens_per_call 3072 \
    --temperature 1.0 --top_p 0.95 --n_sampling 16 --seed 0 \
    --embedding_mode none \
    --output_dir output/stab_A_baseline_qwen1.5b_inst_math500_t1.0_n16
