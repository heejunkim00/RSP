#!/bin/bash
# Evaluate one or more verl FSDP checkpoints on 5 math benchmarks:
#   GSM8K, MATH-500, College Math, Minerva Math, AIME 2024
#
# Convert each FSDP checkpoint to HF format, then run inference + grading
# via the simplerl_math_eval pipeline. Sampling defaults follow the
# convention used to produce the headline numbers in the paper:
#   - aime24            : temperature=1.0, n_sampling=32  (Avg@32)
#   - the other 4       : temperature=0   (greedy), n_sampling=1
#
# Usage:
#   conda activate rsp-dapo
#   bash eval/eval_dapo.sh \
#       --run_name dapo_qwen25_math_7b_baseline \
#       --init_model Qwen/Qwen2.5-Math-7B
#
# Optional flags:
#   --tp_size 1                  vLLM tensor parallel size
#   --benchmarks gsm8k,math500,college_math,minerva_math,aime24
#   --temperature 0              (only used for the 4 greedy benchmarks)
#   --temperature_special 1.0    (only used for aime24)
#   --top_p 1.0
#   --top_p_special 0.95
#   --n_sampling 1               (greedy benchmarks)
#   --n_sampling_special 32      (aime24)
#   --max_tokens 16384
#   --add_step_0 true            also include step_0 (= base model) in eval
#   --specific_steps 10,20,30    only these steps (default: all global_step_*)
#   --overwrite false
#   --output_dir eval_results
#   --wandb_project ""           push to wandb if non-empty
#   --wandb_run_name ""

set -ex

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVAL_PIPE="${REPO_ROOT}/eval/simplerl_math_eval"

# ---------- defaults ----------
TOTAL_NODES=${ARNOLD_WORKER_NUM:-1}
CURRENT_NODE=${ARNOLD_ID:-0}

run_name=""
init_model=""
template="qwen-boxed"
tp_size=1
benchmarks="gsm8k,math500,college_math,minerva_math,aime24"
temperature=0
temperature_special=1.0
top_p=1.0
top_p_special=0.95
n_sampling=1
n_sampling_special=32
max_tokens=16384
add_step_0=false
specific_steps=""
overwrite=false
output_dir="eval_results"
wandb_project=""
wandb_run_name=""

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)            run_name="$2";            shift 2 ;;
        --init_model)          init_model="$2";          shift 2 ;;
        --template)            template="$2";            shift 2 ;;
        --tp_size)             tp_size="$2";             shift 2 ;;
        --benchmarks)          benchmarks="$2";          shift 2 ;;
        --temperature)         temperature="$2";         shift 2 ;;
        --temperature_special) temperature_special="$2"; shift 2 ;;
        --top_p)               top_p="$2";               shift 2 ;;
        --top_p_special)       top_p_special="$2";       shift 2 ;;
        --n_sampling)          n_sampling="$2";          shift 2 ;;
        --n_sampling_special)  n_sampling_special="$2";  shift 2 ;;
        --max_tokens)          max_tokens="$2";          shift 2 ;;
        --add_step_0)          add_step_0="$2";          shift 2 ;;
        --specific_steps)      specific_steps="$2";      shift 2 ;;
        --overwrite)           overwrite="$2";           shift 2 ;;
        --output_dir)          output_dir="$2";          shift 2 ;;
        --wandb_project)       wandb_project="$2";       shift 2 ;;
        --wandb_run_name)      wandb_run_name="$2";      shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$run_name" ] || [ -z "$init_model" ]; then
    echo "Usage: --run_name <run_name> --init_model <model_path_or_hf_id>" >&2
    exit 1
fi

# ---------- paths ----------
CKPTS_DIR="${REPO_ROOT}/checkpoints"
EVAL_DATA_DIR="${EVAL_PIPE}/data"
base_checkpoint_path="${CKPTS_DIR}/${run_name}"
init_model_path="${init_model}"   # HF id or local path; passed straight through

# Verify benchmark data is present
for b in $(echo "$benchmarks" | tr ',' ' '); do
    if [ ! -f "${EVAL_DATA_DIR}/${b}/test.jsonl" ]; then
        echo "[ERR] benchmark data missing: ${EVAL_DATA_DIR}/${b}/test.jsonl" >&2
        echo "      run: python eval/download_eval_data.py" >&2
        exit 1
    fi
done

# ---------- step_0: copy base weights into checkpoint dir if requested ----------
if [ "$add_step_0" = true ]; then
    done_file="$base_checkpoint_path/global_step_0/actor/huggingface/.cp_done"
    if [ "$CURRENT_NODE" -eq 0 ] && [ ! -f "$done_file" ]; then
        echo "[INFO] copying base model into global_step_0/actor/huggingface ..."
        mkdir -p "$base_checkpoint_path/global_step_0/actor/huggingface"
        cp -r "$init_model_path"/* "$base_checkpoint_path/global_step_0/actor/huggingface/"
        touch "$done_file"
    elif [ "$CURRENT_NODE" -ne 0 ]; then
        while [ ! -f "$done_file" ]; do sleep 5; done
    fi
fi

# ---------- collect checkpoint list ----------
get_all_checkpoints() {
    local base_path="$1"
    local specific="$2"
    local checkpoints=()
    if [ -n "$specific" ]; then
        IFS=',' read -r -a step_array <<< "$specific"
        for s in "${step_array[@]}"; do
            local d="$base_path/global_step_$s"
            if [ -d "$d" ]; then checkpoints+=("global_step_$s"); fi
        done
    else
        for d in "$base_path"/global_step_*; do
            [ -d "$d" ] && checkpoints+=("$(basename "$d")")
        done
    fi
    [ ${#checkpoints[@]} -eq 0 ] || printf "%s\n" "${checkpoints[@]}" | sort -V
}

readarray -t all_checkpoints < <(get_all_checkpoints "$base_checkpoint_path" "$specific_steps")
total_ckpts=${#all_checkpoints[@]}
if [ $total_ckpts -eq 0 ]; then
    echo "No checkpoints found in $base_checkpoint_path" >&2
    exit 0
fi

# Distribute checkpoints across nodes
declare -a node_checkpoints
for ((i=0; i<${#all_checkpoints[@]}; i++)); do
    if [ $((i % TOTAL_NODES)) -eq $CURRENT_NODE ]; then
        node_checkpoints+=("${all_checkpoints[i]}")
    fi
done
echo "[INFO] this node ($CURRENT_NODE/$TOTAL_NODES) will evaluate ${#node_checkpoints[@]} checkpoints:"
printf '  %s\n' "${node_checkpoints[@]}"
tmp_ckpt_file=$(mktemp)
printf '%s\n' "${node_checkpoints[@]}" > "$tmp_ckpt_file"

# ---------- run conversion + evaluation ----------
cd "$EVAL_PIPE"
chmod +x sh/convert_and_evaluate_gpu_nodes.sh sh/eval.sh

# convert_and_evaluate_gpu_nodes.sh expects two ranges of sampling params; we
# pass the "regular" set (greedy, n=1) and rely on sh/eval.sh's internal
# special-benchmark branch to switch to (t=temperature_special, n=n_sampling_special)
# for aime24/amc23. The script splits benchmarks accordingly.
bash sh/convert_and_evaluate_gpu_nodes.sh \
    "sh/eval.sh" \
    "$base_checkpoint_path" \
    "$init_model_path" \
    "$template" \
    "$benchmarks" \
    "$temperature" \
    "$max_tokens" \
    "$top_p" \
    "$tp_size" \
    "$tmp_ckpt_file" \
    "$output_dir" \
    "$overwrite" \
    "$n_sampling" \
    ""

# ---------- collect results ----------
if [ -n "$wandb_project" ]; then
    python sh/collect_results.py \
        --base_dir  "$base_checkpoint_path/$output_dir" \
        --model_name "$init_model_path" \
        --wandb_project "$wandb_project" \
        --wandb_run_name "${wandb_run_name:-$run_name}" \
        --temperature "$temperature" \
        --benchmarks "$benchmarks" \
        --use_wandb
else
    python sh/collect_results.py \
        --base_dir  "$base_checkpoint_path/$output_dir" \
        --model_name "$init_model_path" \
        --temperature "$temperature" \
        --benchmarks "$benchmarks"
fi

echo "[DONE] eval finished for run_name=$run_name"
