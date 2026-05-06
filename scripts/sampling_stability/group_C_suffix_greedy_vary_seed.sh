#!/bin/bash
# Group C: softprompt diversity only (greedy, vary embedding_seed 0..15)
#   16 deterministic generations, each with a different random softprompt
for ESEED in $(seq 0 15); do
    python eval.py \
        --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
        --data_dir eval/data --data_names math500 --split test \
        --prompt_type qwen25-math-cot \
        --batch_size 32 --max_tokens_per_call 3072 \
        --temperature 0 --top_p 0.95 --n_sampling 1 --seed 0 \
        --embedding_mode suffix --suffix_position before_im_end \
        --num_embedding_tokens 10 \
        --embedding_seed ${ESEED} \
        --output_dir output/stab_C_suffix_qwen1.5b_inst_math500_greedy_eseed${ESEED}
done
