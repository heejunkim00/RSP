#!/bin/bash
set -x

# Argument check
if [ "$#" -gt 14 ]; then
    echo "Usage: $0 <eval_script_path> <base_checkpoint_path> <init_model_path> <template> [benchmarks] [temperature] [max_tokens] [top_p] [tp_size] [ckpt_list_file] [output_dir] [overwrite] [n_sampling] [rsp_args]"
    exit 1
fi

# Parse arguments
eval_script_path=$1
base_checkpoint_path=$2
init_model_path=$3
template=$4
benchmarks=$5
temperature=$6
max_tokens=$7
top_p=$8
tp_size=${9:-1} 
ckpt_list_file=${10:-""} 
output_dir=${11:-"eval_results"}
overwrite=${12:-false}
n_sampling=${13:-1}
rsp_args=${14:-""}
actor_dir="actor"

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
NUM_GPU_GROUPS=$((NUM_GPUS / tp_size))  # Compute number of GPU groups

# Function: copy tokenizer files
copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
    # Create target path if missing
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # Copy each file
    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# Function: list all checkpoints to evaluate, skipping ones already done
get_checkpoints_to_evaluate() {
    local base_path="$1"
    
    if [ -n "$ckpt_list_file" ] && [ -f "$ckpt_list_file" ]; then
        # Read checkpoints from the provided file
        cat "$ckpt_list_file"
    else
        # Original logic for getting all checkpoints
        local checkpoints=()
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")
                checkpoints+=("$step_tag")
            fi
        done
        
        if [ ${#checkpoints[@]} -eq 0 ]; then
            echo ""
        else
            printf "%s\n" "${checkpoints[@]}"
        fi
    fi
}

# Function: process a single checkpoint on the specified GPU
process_checkpoint() {
    local step_tag=$1
    local group_id=$2
    
    # Compute the GPU ID range for this group
    local start_gpu=$((group_id * tp_size))
    local gpu_ids=""
    for ((i=0; i<tp_size; i++)); do
        if [ -n "$gpu_ids" ]; then
            gpu_ids="${gpu_ids},"
        fi
        gpu_ids="${gpu_ids}$((start_gpu + i))"
    done
    
    ckpt_path="$base_checkpoint_path/$step_tag/$actor_dir/huggingface"
    
    echo "Evaluating checkpoint $step_tag on GPUs $gpu_ids" >&2
    
    output_path_new="$base_checkpoint_path/$output_dir/$step_tag"
    mkdir -p "$output_path_new"
    
    CUDA_VISIBLE_DEVICES=$gpu_ids bash "$eval_script_path" ${template} "$ckpt_path" "$output_path_new" "$temperature" "$max_tokens" "$top_p" "$benchmarks" "$overwrite" "$n_sampling" "$rsp_args"
}

# Save current working directory
original_dir=$(pwd)

# Save current working directory
# Main script section
readarray -t checkpoints_to_evaluate < <(get_checkpoints_to_evaluate "$base_checkpoint_path")

if [ ${#checkpoints_to_evaluate[@]} -eq 0 ]; then
    echo "No new checkpoints to evaluate." >&2
    exit 0
fi

# Get checkpoints to evaluate
if [ $((NUM_GPUS % tp_size)) -ne 0 ]; then
    echo "Error: Number of available GPUs ($NUM_GPUS) is not divisible by tp_size ($tp_size)" >&2
    exit 1
fi

echo "Found ${#checkpoints_to_evaluate[@]} checkpoints to evaluate:" >&2
printf '%s\n' "${checkpoints_to_evaluate[@]}" >&2
total_checkpoints=${#checkpoints_to_evaluate[@]}
eval_count=0
# Check that GPU count is divisible by tp_size
for i in "${!checkpoints_to_evaluate[@]}"; do
    group_id=$((i % NUM_GPU_GROUPS))
    step_tag="${checkpoints_to_evaluate[i]}"
    
    # Parallel checkpoint processing — assign by GPU group
    process_checkpoint "$step_tag" "$group_id" &
    
    # Launch the processing task in background
    if [ $(((i + 1) % NUM_GPU_GROUPS)) -eq 0 ]; then
        wait
    fi
    eval_count=$((eval_count + 1))
    echo "Evaluating $eval_count/$total_checkpoints checkpoints ..."
done

# After launching NUM_GPU_GROUPS tasks, wait for them
wait

cd "$original_dir"
echo "All conversions and evaluations completed."