#!/bin/bash
# Rejection-sampling SFT with Qwen2.5-Math-7B-Instruct.
#
#   bash sft/run_sft.sh rsp 42          # collect with RSP, then SFT and evaluate
#   bash sft/run_sft.sh baseline 42     # collect without RSP
#
# Steps: (1) 8 rollouts per hard problem, one process per GPU on a disjoint shard
#        (2) SFT on the correct rollouts, 1 epoch, effective batch 32
#        (3) evaluate without RSP: greedy on Minerva / GaoKao2023-EN / OlympiadBench,
#            Avg@32 on AIME24 / AMC23
# Env: GPUS (default 0-7), MODEL, HARD_SET, OUT_DIR, GPU_MEM_UTIL (vLLM, default 0.7),
#      NUM_TEST_SAMPLE (smoke test)
set -e
cd "$(dirname "$0")/.."

ARM="${1:?usage: run_sft.sh <rsp|baseline> <seed>}"
SEED="${2:?usage: run_sft.sh <rsp|baseline> <seed>}"
[ "$ARM" = "rsp" ] && RSP_FLAG="--rsp" || RSP_FLAG=""

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
HARD_SET="${HARD_SET:-sft/data/hard_set_qwen25_math_7b_inst.jsonl.gz}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
OUT="${OUT_DIR:-output/sft}/${ARM}_seed${SEED}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
CORRECT="$OUT/correct_${ARM}.jsonl"
mkdir -p "$OUT"

# (1) collection, sharded across GPUs
if [ ! -s "$CORRECT" ]; then
  for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} python sft/collect_rollout.py collect \
      --model "$MODEL" --hard_set "$HARD_SET" --output_dir "$OUT/collect" \
      --n 8 $RSP_FLAG --seed "$SEED" --shard "$i/$NGPU" --gpu_memory_utilization "$GPU_MEM_UTIL" \
      > "$OUT/collect_shard$i.log" 2>&1 &
  done
  wait
  for i in "${!GPU_ARR[@]}"; do
    [ -f "$OUT/collect/correct_${ARM}.shard${i}of${NGPU}.jsonl" ] || { echo "shard $i failed, see $OUT/collect_shard$i.log"; exit 1; }
  done
  cat "$OUT"/collect/correct_${ARM}.shard*.jsonl > "$CORRECT"
  python sft/collection_stats.py --hard_set "$HARD_SET" --correct "$CORRECT" --n 8 | tee "$OUT/collection_stats.txt"
fi

# (2) SFT: per-device batch 2 x grad_accum x NGPU = 32
if [ ! -f "$OUT/sft/config.json" ]; then
  torchrun --nproc_per_node="$NGPU" --master_port=$((29500 + SEED % 1000)) sft/sft_train.py \
    --data "$CORRECT" --model "$MODEL" --output_dir "$OUT/sft" \
    --grad_accum $((16 / NGPU)) --max_length 4096 --seed 42
fi

# (3) evaluation
export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
EXTRA=""; [ -n "${NUM_TEST_SAMPLE}" ] && EXTRA="--num_test_sample ${NUM_TEST_SAMPLE}"
python sft/eval_sft.py --model "$OUT/sft" --output_dir "$OUT/eval" $EXTRA \
  --data_names minerva_math,gaokao2023en,olympiadbench --temperature 0
python sft/eval_sft.py --model "$OUT/sft" --output_dir "$OUT/eval" $EXTRA \
  --data_names aime24,amc23 --temperature 1.0 --top_p 0.95 --n_sampling 32
